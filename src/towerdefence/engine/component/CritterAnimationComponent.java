package towerdefence.engine.component;


import org.newdawn.slick.Animation;
import org.newdawn.slick.GameContainer;
import org.newdawn.slick.Graphics;
import org.newdawn.slick.Image;
import org.newdawn.slick.SlickException;
import org.newdawn.slick.SpriteSheet;
import org.newdawn.slick.state.StateBasedGame;

/**
 *
 * Eventually this will manage rendering animations by loading them from sprite sheets
 *
 * @author Jiv Dhaliwal <jivdhaliwal@gmail.com>
 */
public class CritterAnimationComponent extends RenderComponent {

    // Critter types
    public final static int NORMAL = 0;
    public final static int LEVEL1 = 1;
    public final static int LEVEL2 = 2;
    public final static int BOSS = 3;

    public final static int UP = 0;
    public final static int DOWN = 1;
    public final static int LEFT = 2;
    public final static int RIGHT = 3;

    private final int critterType;

    Image normalSheet = new Image("data/sprites/critters/antNormal.png");

    SpriteSheet critterSheet = new SpriteSheet(normalSheet, 64, 64);
    
    Image[] movementLeft = {critterSheet.getSprite(0, 0), critterSheet.getSprite(1, 0),
        critterSheet.getSprite(2, 0), critterSheet.getSprite(3, 0),
        critterSheet.getSprite(4, 0), critterSheet.getSprite(5, 0),
        critterSheet.getSprite(6, 0), critterSheet.getSprite(7, 0)};
    Image[] movementUp = {critterSheet.getSprite(0, 2), critterSheet.getSprite(1, 2),
        critterSheet.getSprite(2, 2), critterSheet.getSprite(3, 2),
        critterSheet.getSprite(4, 2), critterSheet.getSprite(5, 2),
        critterSheet.getSprite(6, 2), critterSheet.getSprite(7, 2)};
    Image[] movementRight = {critterSheet.getSprite(0, 4), critterSheet.getSprite(1, 4),
        critterSheet.getSprite(2, 4), critterSheet.getSprite(3, 4),
        critterSheet.getSprite(4, 4), critterSheet.getSprite(5, 4),
        critterSheet.getSprite(6, 4), critterSheet.getSprite(7, 4)};
    Image[] movementDown = {critterSheet.getSprite(0, 6), critterSheet.getSprite(1, 6),
        critterSheet.getSprite(2, 6), critterSheet.getSprite(3, 6),
        critterSheet.getSprite(4, 6), critterSheet.getSprite(5, 6),
        critterSheet.getSprite(6, 6), critterSheet.getSprite(7, 6)};
    

    private Animation sprite, up,down,left,right;

    public CritterAnimationComponent(String id, int critterType) throws SlickException
    {
        super(id);
        this.critterType = critterType;

        left = new Animation(movementLeft, 50,true);
        right = new Animation(movementRight, 50,true);
        up = new Animation(movementUp, 50,true);
        down = new Animation(movementDown, 50,true);

        sprite = left;

    }

    @Override
    public void render(GameContainer gc, StateBasedGame sb, Graphics gr) {
        sprite.draw(entity.getPosition().x-16, entity.getPosition().y-32);

    }

    @Override
    public void update(GameContainer gc, StateBasedGame sb, int delta) {
        
        if (entity.getDirection() == LEFT) {
            sprite = left;
        }
        if (entity.getDirection() == RIGHT) {
            sprite = right;
        }
        if (entity.getDirection() == UP) {
            sprite = up;
        }
        if (entity.getDirection() == DOWN) {
            sprite = down;
        }

         
    }


}
